import React from 'react';
import { Modal, Button, Form } from 'react-bootstrap';

const Update = ({show, setShow}) => {
  const handleClose = () => setShow(false);

  const handleSubscribe = (event) => {
    event.preventDefault();
    // Send newsletter subscription request to server

    handleClose();
  };

  return (
    <>
      <Modal show={show} onHide={handleClose}>
        <Modal.Header closeButton>
          <Modal.Title>Subscribe to our Newsletter</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <p>Want <b>updates</b>? Get the latest AI news and updates straight to your inbox.</p>
          <p>Rest assured, esteemed reader, we will <i>never</i> betray your trust by peddling your email to the highest bidder or divulging it to unsavory third parties. We'll only use your email to send you the latest AI news and updates. That's the <b>AI Watch</b> guarantee.</p>
          <Form onSubmit={handleSubscribe}>
            <Form.Group controlId="newsletterForm.Email">
              <Form.Control type="email" placeholder="Enter your email" />
            </Form.Group>
            <Button variant="primary" type="submit" className="mt-4">
              Subscribe
            </Button>
            <Button className="mt-4 ms-2" onClick={() => setShow(false)}>Cancel</Button>
          </Form>
        </Modal.Body>
      </Modal>
    </>
  );
};

export default Update;
